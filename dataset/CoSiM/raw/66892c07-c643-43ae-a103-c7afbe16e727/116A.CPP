#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int entry = 0, exit = 0, inputentry, inputexit, min = 0;
  while (n--) {
    cin >> inputexit >> inputentry;

    entry -= inputexit;
    entry += inputentry;
    if (entry > min)
      min = entry;
  }
  cout << min;
}