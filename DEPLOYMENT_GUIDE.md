# Quarto Book Deployment Guide

## Quick Start

This Quarto book is automatically deployed when you push to the `main` branch via GitHub Actions.

---

## Local Development

### Build the book locally

```bash
cd book
quarto render
```

This generates the HTML output in `book/_book/`.

### Preview the book

```bash
cd book
quarto preview
```

This starts a local development server at `http://localhost:3000`.

---

## Automated Deployment (GitHub Actions)

The workflow file `.github/workflows/quarto-publish.yml` automatically:

1. **Checks** for changes in the `book/` directory
2. **Deploys** pre-rendered HTML from `_book/` to GitHub Pages on the `gh-pages` branch
3. **Serves** live at GitHub Pages

### Requirements

1. **GitHub Pages enabled**:
   - Go to repository **Settings** → **Pages**
   - Source: Deploy from a branch
   - Branch: `gh-pages`
   - Folder: `/ (root)`

### How It Works

Every time you push to `main`:
- GitHub Actions runs the deployment workflow
- Pre-rendered HTML from `book/_book/` is deployed to `gh-pages` branch
- Live at: `https://sanjaykshetri.github.io/tentacles-of-misinformation/`

---

## Manual Deployment

If you prefer manual deployment:

1. **Render locally**:
   ```bash
   cd book
   quarto render
   ```

2. **Verify output**:
   ```bash
   # Check that _book/ has index.html and all chapters
   ls -la book/_book/
   ```

3. **Commit and push**:
   ```bash
   git add book/_book/
   git commit -m "Deploy Quarto book"
   git push origin main
   ```

4. **Manually deploy to Pages**:
   - Push the `_book/` folder contents to the `gh-pages` branch
   - Or use `quarto publish gh-pages` (if Quarto CLI supports it)

---

## Troubleshooting

### Build fails: "Quarto not found"
- Ensure Quarto is installed: `quarto --version`
- Check `.github/workflows/quarto-publish.yml` uses `quarto-dev/quarto-actions/setup@v2`

### Book doesn't appear live
- Wait 60 seconds for GitHub Pages to update
- Check repository **Settings** → **Pages** is configured correctly
- Verify `_book/index.html` exists in the rendered output
- Check **Actions** tab for workflow failures

### Missing chapters or incomplete rendering
- Verify all `.qmd` files in `chapters/` are referenced in `_quarto.yml`
- Check for syntax errors in `.qmd` files
- Run `quarto render --check` locally first

### Custom domain not working
- Ensure DNS A records point to GitHub Pages IP addresses
- Add `CNAME` file to `_book/` with your domain name (GitHub Actions does this automatically if configured)

---

## Configuration

### Domain name
Edit `.github/workflows/quarto-publish.yml` line 53:
```yaml
cname: your-domain.com
```

### Disable PDF output
The current `_quarto.yml` is HTML-only for faster builds. To add PDF:
1. Add PDF format to `_quarto.yml`
2. Ensure LaTeX dependencies are installed in GitHub Actions

### Advanced Rendering
Modify `_quarto.yml` in the `book/` directory to customize:
- Theme
- Navigation  
- Code highlighting
- Bibliography style

---

## Next Steps

1. Verify `tentacles-of-misinformation.com` domain setup
2. Enable GitHub Pages in repository settings
3. Commit and push changes to trigger first deployment
4. Visit your book URL to confirm it's live

For questions, see [Quarto Publishing Docs](https://quarto.org/docs/publishing/github-pages.html)
