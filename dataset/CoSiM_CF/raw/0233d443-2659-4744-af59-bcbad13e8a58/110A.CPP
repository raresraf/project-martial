#include <bits/stdc++.h>

using namespace std;

int main() {
  string number;
  int count = 0;
  int i;
  cin >> number;
  for (i = 0; i <= number.length() - 1; i++) {
    if (number[i] == '4' || number[i] == '7')
      count++;
  }
  if (count == 4 || count == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
