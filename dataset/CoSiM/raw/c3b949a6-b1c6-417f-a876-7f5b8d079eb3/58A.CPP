#include <algorithm>
#include <iostream>
#include <string>
using namespace std;

int main() {

  string aminul;
  // freopen("input.txt","r",stdin);
  cin >> aminul;
  int n = aminul.find('h');
  if (n >= 0) {
    int m = aminul.find('e', n + 1);
    if (m > n) {
      int k = aminul.find('l', m + 1);
      if (k > m) {
        m = aminul.find('l', k + 1);
        if (m > k) {
          k = aminul.find('o', m + 1);
          if (k > m) {
            cout << "YES";
          } else
            cout << "NO";
        } else
          cout << "NO";
      } else
        cout << "NO";
    } else
      cout << "NO";
  } else
    cout << "NO";

  return 0;
}