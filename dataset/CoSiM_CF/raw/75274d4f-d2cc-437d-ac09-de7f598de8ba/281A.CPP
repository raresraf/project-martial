#include <cctype>
#include <iostream>
using namespace std;
/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/
int main() {
  string a;
  cin >> a;
  if (islower(a.at(0))) {
    a.at(0) -= 32;
  }
  cout << a;
  return 0;
}
