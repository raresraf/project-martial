#include <bits/stdc++.h>
using namespace std;

int main() {
  long long int t, i, s, n, mx1, mx2;
  cin >> t;
  while (t--) {
    cin >> n;
    s = 0;
    long long a[n], b[n];
    for (i = 0; i < n; i++)
      cin >> a[i];
    //			if(a[i]<0)
    //				{
    //					b[ngt]==i;
    //					ngt++;
    //				}
    //		}
    //		for(i=0;i<ngt;i++)
    //		{
    //			for(j=b[i]+1;j<b[i+1];j++)
    //				if(a[j+1]>a[j])
    //					mx=a[j+1];
    //			s=s+mx;
    //		}
    //		for(i=0;i<n;i++)
    //			if(a[i]>0)
    //				;
    //			else if(a[i]<0 && a[i+1])
    //
    mx1 = 0;
    mx2 = -1000000001;
    for (i = 0; i < n - 1; i++)
      if (a[i] > 0) {
        mx2 = -1000000001;
        if (a[i] > mx1)
          mx1 = a[i];
        if (a[i + 1] < 0)
          s = s + mx1;
      } else {
        mx1 = 0;
        if (a[i] > mx2)
          mx2 = a[i];
        if (a[i + 1] > 0)
          s = s + mx2;
      }
    if ((a[n - 1] > 0 && a[n - 2] < 0) || (a[n - 1] < 0 && a[n - 2] > 0))
      s = s + a[n - 1];
    else if (a[n - 1] > 0 && a[n - 2] > 0)
      s = s + max(mx1, a[n - 1]);
    else if (a[n - 1] < 0 && a[n - 2] < 0)
      s = s + max(mx2, a[n - 1]);
    if (n == 1)
      s = a[0];
    cout << s << endl;
  }
}
