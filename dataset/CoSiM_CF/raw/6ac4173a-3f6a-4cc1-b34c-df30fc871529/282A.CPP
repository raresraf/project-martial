#include <iostream>
#include <string>
using namespace std;
int main() {
  int n;
  int kq = 0;
  string s;
  cin >> n;
  while (n--) {
    cin >> s;
    char a = s[1];
    if (a == '+') {
      kq++;
    } else {
      kq--;
    }
  }
  cout << kq;
  return 0;
}
