#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {
  string x;
  cin >> x;

  int n = x.length(), i;

  i = count(x.begin(), x.end(), '4') + count(x.begin(), x.end(), '7');

  if (i == 4 | i == 7)
    cout << "YES";

  else
    cout << "NO";
}
