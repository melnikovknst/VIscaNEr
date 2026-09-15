# Uploading a large dataset

Large dataset files in this repository are stored with Git LFS. Put the dataset
inside `data/` or `datasets/`; every file below either directory is tracked by
LFS automatically.

## One-time setup

Install Git LFS, then enable it for your user account:

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

git lfs install
```

Clone the repository after Git LFS is installed:

```bash
git clone https://github.com/melnikovknst/VIscaNEr.git
cd VIscaNEr
```

## Upload the dataset

Copy the files into `data/` (or `datasets/`) and use the normal Git workflow:

```bash
git add data/
git commit -m "Add dataset via Git LFS"
git push origin main
```

Before pushing, verify that the dataset is handled by LFS:

```bash
git lfs ls-files
```

Do not upload the dataset through GitHub's web interface. The person pushing
must have write access to the repository, and the repository owner's GitHub LFS
storage and bandwidth quota must be sufficient for the dataset.
