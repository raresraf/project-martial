#include <iostream>
using namespace std;
int main() {
  int n, counter = 0;
  string stone;
  cin >> n;
  cin >> stone;
  for (int i = 0; i < stone.length() - 1; i++) {
    if (stone[i] == stone[i + 1]) {
      counter++;
    }
  }
  cout << counter << endl;
}
