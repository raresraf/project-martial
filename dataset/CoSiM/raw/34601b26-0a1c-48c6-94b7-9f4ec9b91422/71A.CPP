#include <iostream>
#include <string>
using namespace std;
int strlength(string a) {
  int count = 0;
  while (a[count] != '\0') {
    count++;
  }
  return count;
}
int main() {
  int n, no_of_testcases;
  cin >> no_of_testcases;
  for (n = 0; n < no_of_testcases; n++) {
    string a;
    int length;
    cin >> a;
    length = strlength(a);
    if (length > 10)
      cout << a[0] << length - 2 << a[length - 1] << endl;
    else
      cout << a << endl;
  }
  return 0;
}
