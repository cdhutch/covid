# covid

This Python script creates graphs showing growth rate of COVID-19 in various US states in the style of the [Financial Times](https://www.ft.com/coronavirus-latest).

The source of the data is the [COVID Tracking Project](https://covidtracking.com).

The script generates two `.pdf` files, once showing number of positive cases and the other showing number of deaths. In theory, the script will also launch the default PDF viewer for your operating system. If the script returns an error, add a `#` symbol before the line that reads `open_pdf(fname)` and rerun.`