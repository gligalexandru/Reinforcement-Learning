import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class QLearnerBothHandsHoldAce {
    private static final double LEARNING_RATE = 0.1;
    private static final double DISCOUNT_FACTOR = 0.9;
    private static final int NUM_EPISODES = 1000000;

    //dealer_hand_sum->own_hand_sum->action->value
    private static HashMap<Boolean, HashMap<Integer, HashMap< Integer, HashMap<Integer, Double>>>> QTable;

    public static void main(String[] args) {
        BlackJackEnv game = new BlackJackEnv(BlackJackEnv.NONE);
        // Initialize QTable
        QTable = new HashMap<>();
        // Variables to measure and report average performance
        double totalReward = 0.0;
        int numberOfGames = 0;

        double wins = 0;
        double losses = 0;
        double draw = 0;

        while (notDone()) {
            double currentReward = playOneGame(game);
            totalReward += currentReward;

            if (currentReward == 1) {
                wins += 1;
            } else if (currentReward == 0) {
                draw += 1;
            } else if (currentReward == -1) {
                losses += 1;
            } else {
                int a = 1/0;
            }

            numberOfGames++;
            if ((numberOfGames % 1000) == 0) {
                System.out.println("Games played: " + numberOfGames);
                System.out.println("Number of wins in last 1000 games: " + wins);
                System.out.println("Number of losses in last 1000 games: " + losses);
                System.out.println("Number of draws in last 1000 games: " + draw);
                System.out.println("Win ratio in last 1000 games: " + (wins / (losses + draw + wins)));
                System.out.println("----------------------------");

                wins = 0;
                losses = 0;
                draw = 0;
            }
        }
        // Show the learned QTable
        outputQTable();
    }

    private static double playOneGame(BlackJackEnv game) {
        ArrayList<String> state = game.reset();
        double totalReward = 0.0;
        Random rand = new Random();

        while (!Boolean.parseBoolean(state.get(0))) {
            int action = selectAction(state, rand);
            ArrayList<String> nextState = game.step(action);
            double reward = Double.parseDouble(nextState.get(1));
            totalReward += reward;

            // Update Q-value based on Q-learning formula
            updateQValue(state, action, nextState, reward);

            state = nextState;
        }

        return totalReward;
    }

    private static int selectAction(ArrayList<String> state, Random rand) {
        // This is just an example, you may choose different exploration-exploitation strategies
        // Epsilon-greedy approach
        double epsilon = 0.1; // Exploration probability
        Boolean hold_active_ace = BlackJackEnv.holdActiveAce(BlackJackEnv.getPlayerCards(state));
        Integer current_own_hand_value = BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(state));
        Integer current_dealer_hand_value = BlackJackEnv.totalValue(BlackJackEnv.getDealerCards(state));

        if (    !QTable.containsKey(hold_active_ace) ||
                !QTable.get(hold_active_ace).containsKey(current_dealer_hand_value) ||
                !QTable.get(hold_active_ace).get(current_dealer_hand_value).containsKey(current_own_hand_value) ||
                rand.nextDouble() < epsilon) {
            // Random action
            return rand.nextInt(2);
        } else {
            // Greedy action
            int bestAction = 0;
            double bestQValue = QTable.get(hold_active_ace)
                    .get(current_dealer_hand_value)
                    .get(current_own_hand_value)
                    .getOrDefault(0, 0.0); // Default to 0.0 if Q-value is not present
            for (int action : QTable
                    .get(hold_active_ace).get(current_dealer_hand_value).get(current_own_hand_value).keySet()) {
                double qValue = QTable
                        .get(hold_active_ace).get(current_dealer_hand_value).get(current_own_hand_value)
                        .getOrDefault(action, 0.0); // Default to 0.0 if Q-value is not present
                if (qValue > bestQValue) {
                    bestQValue = qValue;
                    bestAction = action;
                }
            }
            return bestAction;
        }
    }

    private static void updateQValue(ArrayList<String> state, int action, ArrayList<String> nextState, double reward) {
        // Initialize Q-values for the current state if not already present
        Boolean hold_active_ace = BlackJackEnv.holdActiveAce(BlackJackEnv.getPlayerCards(state));
        Integer current_own_hand_value = BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(state));
        Integer current_dealer_hand_value = BlackJackEnv.totalValue(BlackJackEnv.getDealerCards(state));

        QTable.putIfAbsent(hold_active_ace, new HashMap<>());
        QTable.get(hold_active_ace).putIfAbsent(current_dealer_hand_value, new HashMap<>());
        QTable.get(hold_active_ace).get(current_dealer_hand_value).putIfAbsent(current_own_hand_value, new HashMap<>());
        QTable.get(hold_active_ace).get(current_dealer_hand_value).get(current_own_hand_value).putIfAbsent(action, 0.0);

        double currentQValue = QTable
                .get(hold_active_ace).get(current_dealer_hand_value).get(current_own_hand_value).get(action);
        double maxNextQValue = 0.0;

        if (!Boolean.parseBoolean(nextState.get(0))) {
            // Game is not over, get the maximum Q-value for the next state if it exists
            maxNextQValue = QTable
                    .get(hold_active_ace)
                    .get(current_dealer_hand_value)
                    .getOrDefault(BlackJackEnv.totalValue(BlackJackEnv.getPlayerCards(nextState)), new HashMap<>())
                    .values()
                    .stream()
                    .mapToDouble(Double::doubleValue)
                    .max()
                    .orElse(0.0);
        }

        // Q-learning update formula
        double newQValue = currentQValue + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxNextQValue - currentQValue);
        QTable.get(hold_active_ace).get(current_dealer_hand_value).get(current_own_hand_value).put(action, newQValue);
    }

    private static boolean notDone() {
        return episodeCounter++ <= NUM_EPISODES;
    }

    private static void outputQTable() {
        // Output or visualize your QTable
        // Here you can print Q-values or display them in any other form
        System.out.println("QTable: " + QTable);
    }

    private static int episodeCounter = 0;
}
