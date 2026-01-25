#include <iostream>
#include <string>
using namespace std;
int main() {
  int n = 0, count = 0;
  cin >> n;
  char tmp, so, sn;
  cin >> so;
  n = n - 1;
  while (n--) {
    cin >> sn;
    if (so == sn)
      count++;
    so = sn;
  }
  cout << count << endl;
}
