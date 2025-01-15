#include <iostream>
using namespace std;
int main() {
  int stop, i, in, out, max = 0, total = 0;
  cin >> stop;
  for (i = 0; i < stop; i++) {
    cin >> out >> in;
    total += in - out;
    if (max < total)
      max = total;
  }
  cout << max;
  return 0;
}
