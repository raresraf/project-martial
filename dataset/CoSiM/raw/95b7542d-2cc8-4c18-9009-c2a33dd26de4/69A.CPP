#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  int a = 0, b = 0, c = 0;
  int sum1 = 0, sum2 = 0, sum3 = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b >> c;
    sum1 += a;
    sum2 += b;
    sum3 += c;
  }

  if (sum1 == 0 && sum2 == 0 && sum3 == 0) {
    cout << "YES";
  } else
    cout << "NO";
  return 0;
}
