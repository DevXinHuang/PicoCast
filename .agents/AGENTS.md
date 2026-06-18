# PicoCAST Project Rules

- **Always Deploy Fixes to GitHub Pages**: Whenever a fix, update, or new feature is implemented for the Review Packet Dashboard or any website-related asset, automatically regenerate the final HTML files (`build_tracklet_review_packet.py` or `make_review_packet_dashboard.py`), copy them to the `docs/review_packet/` deployment folder, and commit + push to GitHub. This ensures the user can immediately see the live, working result without having to ask for it.
