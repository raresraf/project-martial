#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  vector<int> ar(n);
  for (int &i : ar)
    cin >> i;

  if (n <= 2 or n % 2 != 0)
    cout << "NO";
  else
    cout << "YES";
  cout << 1;
  return 0;
}