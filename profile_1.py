import seam_carving.cli


def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        seam_carving.cli.main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats("profile.stats")


if __name__ == "__main__":
    main()
