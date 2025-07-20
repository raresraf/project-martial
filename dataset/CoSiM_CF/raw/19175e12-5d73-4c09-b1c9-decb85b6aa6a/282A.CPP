#include <iostream>
using namespace std;
/*
    Author- Sohanuzzaman Soad
    University of Asia Pacific (UAP),BD

*/
int main() {
  int n, r = 0;
  cin >> n;

  string soad;
  while (n--) {
    cin >> soad;
    if (soad[1] == '+')
      r++;
    else
      r--;
  }
  cout << r << endl;
  return 0;
}
