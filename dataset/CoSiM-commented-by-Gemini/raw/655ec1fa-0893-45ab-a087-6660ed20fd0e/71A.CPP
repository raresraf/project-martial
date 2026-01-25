// Codeforces
// 71A - Way too long word

#include <iostream>
#include <vector>
using namespace std;

int main() {
  int size;
  cin >> size;

  vector<string> vs;
  string line;

  for (int i = 0; i < size; i++) {
    cin >> line;
    vs.push_back(line);
  }

  for (int i = 0; i < size; i++) {
    if (vs[i].size() <= 10)
      cout << vs[i] << endl;
    else {
      cout << vs[i][0];
      cout << (vs[i].size() - 2);
      cout << vs[i][vs[i].size() - 1] << endl;
    }
  }

  return 0;
}
