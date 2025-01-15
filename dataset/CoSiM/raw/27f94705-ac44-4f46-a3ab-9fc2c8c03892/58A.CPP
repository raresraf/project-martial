#include <bits/stdc++.h>
using namespace std;
int main() {
  string x;
  cin >> x;
  int count = 0;
  for (int i = 0; i < x.length(); i++) {
    if (count == 0 && x[i] == 'h')
      count++;
    else if (count == 1 && x[i] == 'e')
      count++;
    else if (count == 2 && x[i] == 'l')
      count++;
    else if (count == 3 && x[i] == 'l')
      count++;
    else if (count == 4 && x[i] == 'o')
      count++;
  }
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
}