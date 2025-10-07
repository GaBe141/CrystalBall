"""Run the Streamlit dashboard.

This helper allows launching the dashboard with:
  python -m src.run_dashboard
"""
import os


def main():
    from streamlit.web import bootstrap
    # Resolve path to dashboard.py
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, 'dashboard.py')
    bootstrap.run(app_path, '', [], flag_options={})

if __name__ == "__main__":
    main()
