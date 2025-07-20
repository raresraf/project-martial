#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int answer = 0;
  while (n--) {
    string operation;
    cin >> operation;
    if (operation.find('+') != std::string::npos) {
      answer++;
    } else {
      answer--;
    }
  }
  cout << answer;
  return 0;
}
