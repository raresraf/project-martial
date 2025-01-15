#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string a, b;
  getline(cin, a);
  getline(cin, b);

  transform(a.begin(), a.end(), a.begin(), ::tolower);
  transform(b.begin(), b.end(), b.begin(), ::tolower);

  if (a == b)
    cout << '0';
  else if (a < b)
    cout << "-1";
  else
    cout << '1';

  return 0;
}
