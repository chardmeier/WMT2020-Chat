import argparse
import curses
import random
import scipy.stats
import textwrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sys1')
    parser.add_argument('sys2')
    args = parser.parse_args()

    sys = []

    with open(args.sys1, 'r') as f:
        sys.append([line.rstrip('\n') for line in f])

    with open(args.sys2, 'r') as f:
        sys.append([line.rstrip('\n') for line in f])

    if len(sys[0]) != len(sys[1]):
        raise ValueError('Unequal file length.')

    counts = eval_systems(sys)
    print('System 1 wins:', counts[0])
    print('System 2 wins:', counts[1])
    print('Ties:         ', counts[2])

    p = scipy.stats.binom_test(counts[:2])
    if p >= .05:
        print('Sign test: Comparison not significant (p = %g)' % p)
    else:
        rank = [sys for _, sys in sorted(zip(counts[:2], [1, 2]), reverse=True)]
        print('Sign test: System %d > System %d (p = %g)' % (rank[0], rank[1], p))


def eval_systems(sys):
    return curses.wrapper(do_eval, sys)


def do_eval(stdscr, sys):
    curses.curs_set(False)
    stdscr.clear()
    _, linewidth = stdscr.getmaxyx()

    nex = len(sys[0])
    lines_per_system = 5

    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_MAGENTA, -1)

    wtrans1 = stdscr.derwin(lines_per_system + 1, linewidth, 0, 0)
    wtrans1.attron(curses.color_pair(1))
    wtrans1.addstr(0, 0, 'Translation 1:', curses.color_pair(1) | curses.A_BOLD)
    wtrans1_txt = wtrans1.derwin(1, 0)

    wtrans2 = stdscr.derwin(lines_per_system + 1, linewidth, lines_per_system + 2, 0)
    wtrans2.attron(curses.color_pair(2))
    wtrans2.addstr(0, 0, 'Translation 2:', curses.color_pair(2) | curses.A_BOLD)
    wtrans2_txt = wtrans2.derwin(1, 0)

    stdscr.refresh()

    counts = [0, 0, 0]

    for i in random.sample(range(nex), nex):
        first = random.randrange(2)
        second = 1 - first

        wrapped_first = textwrap.wrap(sys[first][i], linewidth)
        wrapped_second = textwrap.wrap(sys[second][i], linewidth)

        wtrans1_txt.clear()
        for j, l in enumerate(wrapped_first[:lines_per_system]):
            wtrans1_txt.addstr(j, 0, l)

        wtrans2_txt.clear()
        for j, l in enumerate(wrapped_second[:lines_per_system]):
            wtrans2_txt.addstr(j, 0, l)

        wtrans1_txt.refresh()
        wtrans2_txt.refresh()

        while True:
            key = stdscr.getkey()
            if key == 'q':
                return counts
            elif key == '1':
                counts[first] += 1
                break
            elif key == '2':
                counts[second] += 1
                break
            elif key == ' ':
                counts[-1] += 1
                break

    return counts


if __name__ == '__main__':
    main()
