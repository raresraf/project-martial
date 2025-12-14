#include <bits/stdc++.h>
using namespace std;

int main() {
  string num;
  int counter = 0, i;
  cin >> num;
  for (i = 0; i < num.length(); i++) {
    if (num[i] == '4' || num[i] == '7')
      counter++;
  }
  if (counter == 4 || counter == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
