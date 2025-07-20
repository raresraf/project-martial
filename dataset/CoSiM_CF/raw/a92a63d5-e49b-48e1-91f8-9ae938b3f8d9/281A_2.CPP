#include <bits/stdc++.h>
using namespace std;
int main() {
  /* code */
  string x;
  cin >> x;
  if (islower(x.at(0)))
    x.at(0) = toupper(x.at(0));
  cout << x;
  return 0;
}