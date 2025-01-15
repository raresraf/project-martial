#include <iostream>
using namespace std;
int main() {
  int a, i;
  cin >> a;
  float sum = 0;
  int b[a];
  for (i = 0; i < a; i++) {
    cin >> b[i];
    sum += b[i];
  }
  cout << sum / a;
}
