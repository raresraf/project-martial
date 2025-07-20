#include <iostream>
using namespace std;
int main() {
  int incoming, outgoing;
  int pass = 0, temp = 0;
  int nstn;
  cin >> nstn;
  for (int i = 0; i < nstn; i++) {
    cin >> outgoing >> incoming;
    temp = temp + incoming - outgoing;
    if (pass < temp)
      pass = temp;
  }
  cout << pass;
  return 0;
}
