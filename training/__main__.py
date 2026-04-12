"""Allow `python -m training` as shortcut for `python -m training.selfplay loop`."""

from training.selfplay import main

if __name__ == '__main__':
    main()
