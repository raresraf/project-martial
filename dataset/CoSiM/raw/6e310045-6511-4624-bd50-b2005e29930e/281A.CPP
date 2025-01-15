#include <iostream>
#include <string.h>

using namespace std;

int main() {
  string a;
  cin >> a;
  if (a[0] >= 97 && a[0] <= 122) // checking if it is lower case.
    a[0] -= 32;
  cout << a;

  return 0;
}