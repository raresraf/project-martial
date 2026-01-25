#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;
#define gc getchar_unlocked
inline int getchar_unlocked() { return getchar(); }
void scanint(int &x);

void scanint(int &x) {
  register int c = gc();
  x = 0;
  int neg = 0;
  for (; ((c < 48 || c > 57) && c != '-'); c = gc())
    ;
  if (c == '-') {
    neg = 1;
    c = gc();
  }
  for (; c > 47 && c < 58; c = gc()) {
    x = (x << 1) + (x << 3) + c - 48;
  }
  if (neg)
    x = -x;
}
int main() {
  int n, i, j, sum = 0, taken = 0;
  ;
  vector<int> a;
  scanint(n);
  for (i = 0; i < n; i++) {
    scanint(j);
    sum += j;
    a.push_back(j);
  }
  sort(a.rbegin(), a.rend());
  // for(i=0; i<n; i++)
  //	cout<<a[i]<<endl;
  i = 0;
  while (taken <= sum) {
    taken += a[i];
    sum = sum - a[i];
    i++;
  }
  // cout<<"sum "<<sum<<"taken "<<taken<<" "<<i<<endl;
  cout << i << endl;
  return 0;
}