#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, m, a;
  cin >> n >> m >> a;

  long long countn;

  if (n % a == 0) {
    countn = n / a;

  } else {
    countn = n / a;
    countn++;
  }

  long long flagsByColumn;

  if (m % a == 0) {
    countm = m / a;

  } else {
    countm = m / a;
    countm++;
  }

  cout << countn * countm << endl;

  return 0;
}
