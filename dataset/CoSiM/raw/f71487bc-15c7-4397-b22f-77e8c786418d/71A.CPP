#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

int main(void) {
  int lineCount;
  string line;

  cin >> lineCount;
  for (int i = 0; i < lineCount; i++) {
    cin >> line;
    size_t length = line.length();
    if (length > 10) {
      line = line[0] + to_string(length - 2) + line[length - 1];
    }
    cout << line << endl;
  }

  return 0;
}