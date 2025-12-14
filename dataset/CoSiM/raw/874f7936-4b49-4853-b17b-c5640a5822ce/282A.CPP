#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    string x;
    cin >> x;
    if (x[0] == '+' && x[1] == '+')
      count++;
    else if (x[0] == '-' || x[1] == '-')
      count--;
    else if (x[1] == '+' && x[2] == '+')
      count++;
    else if (x[1] == '-' && x[2] == '-')
      count--;
  }
  cout << count << endl;
  return 0;
}