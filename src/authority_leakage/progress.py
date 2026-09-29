"""Optional import fallback; tqdm is declared as a project dependency."""
try:
    from tqdm.auto import tqdm
except ImportError:  # supports dataset inspection before dependencies are synced
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class NullProgress:
                def update(self, amount=1):
                    pass
                def close(self):
                    pass
            return NullProgress()
        return iterable
