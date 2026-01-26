#include <iostream>
using namespace std;
int main() {

  long long n, rst = 0;
  int k, w;
  cin >> k >> n >> w;
  for (int i = 1; i <= w; i++) {
    rst += k * i;
  }
  if (rst - n > 0)
    cout << rst - n;
  else
    cout << 0;
}
