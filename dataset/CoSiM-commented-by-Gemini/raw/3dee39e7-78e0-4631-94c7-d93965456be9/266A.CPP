#include <cctype>
#include <iostream>
using namespace std;
/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/
int main() {
  int n;
  cin >> n;
  string a;
  cin >> a;
  int red = 0;
  int green = 0;
  int blue = 0;
  for (int i = 1; i < n; i++) {
    if (a[i - 1] == a[i])
      if (a[i] == 'R') {
        ++red;
      } else if (a[i] == 'G') {
        ++green;
      } else if (a[i] == 'B') {
        ++blue;
      }
  }
  cout << red + green + blue;
  return 0;
}
