#include <iostream>
using namespace std;
int main() {
  int n, a, b, temp = 0, maxx = 0;
  cin >> n;
  while (n--) {
    cin >> a >> b;
    temp = temp - a + b;
    if (temp > maxx)
      maxx = temp;
  }
  cout << maxx << endl;
  return 0;
}
