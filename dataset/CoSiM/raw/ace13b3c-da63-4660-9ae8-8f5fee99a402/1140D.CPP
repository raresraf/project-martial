#include <bits/stdc++.h>

using namespace std;

int main() {

  int n;
  cin >> n;
  long long int prod = 0, i = 3;
  while (i <= n) {
    prod += (i) * (i - 1);
    i++;
  }

  cout << prod << endl;

  return 0;
}
