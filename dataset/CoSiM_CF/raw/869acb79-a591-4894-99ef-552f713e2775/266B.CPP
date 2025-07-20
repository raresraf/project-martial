#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main() {
  int n, t;
  string kata;
  vector<int> list;
  cin >> n >> t;
  cin >> kata;

  for (int i = 0; i < t; i++) {
    for (int j = 1; j < kata.length(); j++) {
      if (kata[j] == 'G' && kata[j - 1] == 'B') {
        list.push_back(j);
      }
    }
    for (int j = 0; j < list.size(); j++) {
      kata[list[j] - 1] = 'G';
      kata[list[j]] = 'B';
    }
    list.clear();
  }
  cout << kata;
  return 0;
}
