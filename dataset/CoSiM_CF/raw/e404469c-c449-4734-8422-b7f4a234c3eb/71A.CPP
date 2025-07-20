#include <bits/stdc++.h>

using namespace std;

#define MAX_LENGTH 10

int main() {

  int n;
  cin >> n;

  string x;
  while (n--) {
    cin >> x;

    int s = x.length();

    if (s > MAX_LENGTH) {
      cout << x[0] << s - 2 << x[s - 1] << "\n";
    } else {
      cout << x << "\n";
    }
  }
}