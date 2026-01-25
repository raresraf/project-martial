#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;

  int total = 0;

  while (n--) {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);

    int sum = a + b + c;

    if (sum >= 2) {
      total++;
    }
  }
  cout << total << endl;

  return 0;
}
