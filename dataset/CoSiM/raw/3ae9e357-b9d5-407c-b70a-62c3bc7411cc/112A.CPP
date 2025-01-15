#include <bits/stdc++.h>
using namespace std;

int main() {
  string firstString;
  string secondString;

  cin >> firstString;
  cin >> secondString;

  transform(firstString.begin(), firstString.end(), firstString.begin(),
            ::tolower);
  transform(secondString.begin(), secondString.end(), secondString.begin(),
            ::tolower);

  if (firstString > secondString)
    cout << 1;
  else if (secondString > firstString)
    cout << -1;
  else
    cout << 0;

  return 0;
}
