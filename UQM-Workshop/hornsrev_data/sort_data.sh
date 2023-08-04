awk -v COLT=$1 '
        NR==1 {
                for (i=1; i<=NF; i++) {
                        if ($i==COLT) {
                                title=i;
                                print $i;
                        }
                }
        }
        NR>1 {
                if (i=title) {
                        print $i;
                }
        }
' hornsrev_268p00_8p00_report.txt