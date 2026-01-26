#include "bits/stdc++.h"

using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;

  int a, b;

  int max = 0;
  int sum = 0;

  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    sum = sum - a + b;
    if (sum > max)
      max = sum;
  }

  cout << max;
  return 0;
}
