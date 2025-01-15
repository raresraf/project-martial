#include <bits/stdc++.h>
using namespace std;
main() {
  int n, k, m;
  cin >> n >> k;
  int a[n];
  int count = 0;
  k = k - 1;
  for (int i = 0; i < n; i++) {
    cin >> a[i];
    //	    if(a[i]==0)
    //	      count++;

    if (i == k)
      m = a[i];
  }
  //	   if(i>=k)
  //	    {
  //	      if(a[i]<a[k])
  //		    count++;
  //	    }
  //	    if(i>=k && a[i]<=0)
  //	      count++;
  //	}
  //	cout<<n-count;

  for (int i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= m)
      count++;
  }
  cout << count;
}
