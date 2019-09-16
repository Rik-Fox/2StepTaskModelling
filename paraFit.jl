using Distributed
  # Now add 2 procs that can exec in parallel (obviously it depends on your CPU
  # what you actually gain from this though)
  #addprocs(2)

  # Ensure BlackBoxOptim loaded on all workers
  @everywhere using BlackBoxOptim, AgentTreeModels

  @everywhere open("Results/res_daw.csv", "w") do io
      @parallel for i=1:112
          cleanData = groupby(DataFrame(CSV.File("/home/rfox/Project_MSc/data/Subj$i.csv", delim = ',')), :Flex0_or_Spec1)[1]
          exData = [t == 1 for t in [cleanData.First_Choice [t == 0.3 for t in cleanData.Transition_Prob] cleanData.Second_Choice cleanData.Reward]]

          open("Data/data_daw.csv","w") do io
              writedlm(io, exData)
          end

          bbopt = @timed bboptimize(log_likelihood, SearchRange=[(0.0, 1.0),(2.0, 25.0),(2.0,25.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)],workers())

          writedlm(io, [best_candidate(bbopt[1]),bbopt[2]])
          end
      end
  end
