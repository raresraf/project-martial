#include <iostream>
#include <string.h>

using namespace std;

int main() {
  int n, m;
  cin >> n >> m;
  string row;
  cin >> row;
  string newrow = row;
  for (int i = 0; i < m; i++) {
    for (int i = 0; i < n - 1; i++) {
      if (row[i] == 'B' && row[i + 1] == 'G') {
        newrow[i] = 'G';
        newrow[i + 1] = 'B';
      }
    }
    row = newrow;
  }
  cout << newrow << endl;
}
