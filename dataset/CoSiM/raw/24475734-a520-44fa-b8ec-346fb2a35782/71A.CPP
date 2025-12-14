#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
using std::cin;
using std::cout;
using std::ostringstream;
using std::string;
int main() {
  int n;
  string word;
  cin >> n;
  for (int i = 0; i < n; ++i) {
    cin >> word;
    if (word.size() > 10) {
      ostringstream ss;
      ss << (word.size() - 2);
      word = word.front() + ss.str() + word.back();
    }
    cout << word << "\n";
  }
  return 0;
}
