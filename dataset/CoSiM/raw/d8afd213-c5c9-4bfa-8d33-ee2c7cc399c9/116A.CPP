#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  int ai, bi;
  int inTram = 0, max = 0;
  cin >> n;
  while (n--) {
    cin >> ai >> bi;
    inTram += (-ai + bi);
    if (inTram > max)
      max = inTram;
  }

  cout << max << endl;

  return 0;
}