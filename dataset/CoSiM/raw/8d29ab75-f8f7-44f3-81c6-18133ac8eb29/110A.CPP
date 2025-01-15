#include <algorithm>
#include <iostream>
#include <string.h>
using namespace std;

int main() {
  string num;
  cin >> num;

  int seven = count(num.begin(), num.end(), '7');
  int four = count(num.begin(), num.end(), '4');
  if ((four + seven == 4 || four + seven == 7))
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
