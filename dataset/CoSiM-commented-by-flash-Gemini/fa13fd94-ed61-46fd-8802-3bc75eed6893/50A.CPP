#include <iostream>
using namespace std;
int main() {
  long long int a, b, c, d, coun = 0;
  ;
  cin >> d;
  for (int i = 1; i <= d; i++) {
    cin >> a;
    cin >> b;
    cin >> c;
    if (a + b + c >= 2)
      coun++;
  }
  cout << coun++;
}
