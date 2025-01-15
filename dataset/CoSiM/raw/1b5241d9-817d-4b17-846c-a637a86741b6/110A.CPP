#include <bits/stdc++.h>
using namespace std;
int main() {
  long long int n;
  cin >> n;
  int cnt = 0;
  while (n != 0) {
    int t = n % 10;
    if (t == 4 || t == 7)
      cnt++;
    n /= 10;
  }
  if (cnt == 4 || cnt == 7)
    cout << "YES";
  else
    cout << "NO";
}